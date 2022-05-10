import random
import numpy as np
import os 
import time
from multiprocessing import Process


INDEX_MAP = {'A':0,'C':1,'G':2,'T':3}
PRINT_EPOCH = 500
ITER = 500
LOG = False

def construct_PWM(sites):
  ml = len(sites[0])
  sc_1 = len(sites)
  matrix = np.zeros((ml,4))
  for i in range(sc_1):
    for j in range(ml):
      if(sites[i][j] == 'A'): matrix[j][0] += 1
      if(sites[i][j] == 'C'): matrix[j][1] += 1
      if(sites[i][j] == 'G'): matrix[j][2] += 1
      if(sites[i][j] == 'T'): matrix[j][3] += 1

  matrix = matrix / sc_1
  return matrix   

def get_source_nucleotide_distrib(sequences):
  combined_sequence = ''.join(sequences)
  a_prob = combined_sequence.count('A')/len(combined_sequence)
  c_prob = combined_sequence.count('C')/len(combined_sequence)
  g_prob = combined_sequence.count('G')/len(combined_sequence)
  t_prob = combined_sequence.count('T')/len(combined_sequence)

  return {'A': a_prob,'C': c_prob,'G': g_prob,'T': t_prob}

def calculate_icpc(motif):
  s = 0
  for row in motif:
    for nucleotide_prob in row:
      if(nucleotide_prob != 0):
        s += nucleotide_prob * np.log2(nucleotide_prob / 0.25)

  return s/ len(motif)

def gibbs_sampling(sequences, ml):
  '''
  sequences: random sc nucleotide sequnces obtained form benchmark dataset
  ml: motif_length

  Returns:
  predicted_pwm = motif pwm matrix (ml,4)
  predicted_sites = sc indices 
  ''' 
  sc = len(sequences)
  sl = len(sequences[0])
  iter = ITER
  bias = 1e-7
  src_distribution = get_source_nucleotide_distrib(sequences)
  if(LOG): print(f'P(x) = {src_distribution}')

  # 1. select one random index per sequence indicating the start of the binding site
  random_indices = np.random.randint(0, sl - ml + 1, sc)
  if(LOG): print(f'Random indices: {random_indices}')

  # 2. Loop until iter iterations
    # 3. select one binding site sequence and update by constructing the PWM matrix using the remaining sc-1 binding site sequences
  for i in range(iter):
    binding_sites = [sequences[idx][random_indices[idx]:random_indices[idx] + ml] for idx in range(sc)]
    index_to_leave = (i % sc)
    binding_sites_to_consider = [binding_sites[i] for i in range(sc) if(i != index_to_leave)]
    matrix_PWM = construct_PWM(binding_sites_to_consider)

    curr_binding_indices = []
    scores = []
    for j in range(sl - ml + 1):
      curr_binding_site = sequences[index_to_leave][j:j + ml] 
      p_x, q_x = 1, 1
      for index, nucleotide in enumerate(curr_binding_site):
        q_x *=  matrix_PWM[index][INDEX_MAP[nucleotide]]
        p_x *= src_distribution[nucleotide]

      score = q_x / (p_x + bias)

      curr_binding_indices.append(j)
      scores.append(score)
    
    scores = np.array(scores)
      
    if(np.sum(scores)!=0):
      scores = scores / np.sum(scores)  
      updated_binding_index =  np.random.choice(curr_binding_indices,1,p=scores) # curr_binding_indices[np.argmax(scores)]
      random_indices[index_to_leave] = updated_binding_index 
    
    if(LOG and i%PRINT_EPOCH == 0):
      print(f'ITER: {i}, length: {len(scores)}, icpc: {calculate_icpc(matrix_PWM)}, updated_sites: {random_indices}')

  binding_sites = [sequences[idx][random_indices[idx]:random_indices[idx] + ml] for idx in range(sc)]
  final_PWM = construct_PWM(binding_sites)

  return final_PWM, random_indices

def motif_finding(dataset_path):
  '''
  Runs gibbs sampling on one dataset for ITER iterations and returns run time
  '''
  
  sequences = []
  ml = 0
  with open(dataset_path +'/sequences.fa') as fp:
    lines = fp.readlines()
    for line in lines:
      if('>' not in line):
        sequences.append(line.strip())
        assert len(line.strip()) == 500

  with open(dataset_path+'/motiflength.txt') as fp:
    ml = int(fp.read().strip())
  
  start_time = time.time()
  predicted_pwm, predicted_sites = gibbs_sampling(sequences, ml)
  end_time = time.time()
  if(LOG): 
    print(f'Predicted PWM:\n {predicted_pwm}')
    print(f'Predicted Sites:\n {predicted_sites}')
  
  return end_time - start_time, predicted_pwm, predicted_sites

def store_motif_finding_results(dataset_path, predicted_sites, predicted_pwm, best_run_time):
  ml = 0
  with open(dataset_path+'/motiflength.txt') as fp:
    ml = int(fp.read().strip())

  with open(dataset_path + '/predictedsites.txt', 'w+') as fp:
    fp.writelines([str(index) + '\n' for index in predicted_sites])

  with open(dataset_path + '/predictedmotif.txt', 'w+') as fp:
    doc_text = '>PREDICTEDMOTIF_1\t' + str(ml) + '\n'
    for i in range(ml):
      doc_text += '\t'.join([str(val) for val in predicted_pwm[i]]) + '\n'
    
    doc_text += '<'
    fp.write(doc_text)
  
  with open(dataset_path + '/avgruntime.txt', 'w+') as fp:
    fp.write(str(best_run_time))


def proc(dataset_path):
  best_score, best_run_time, best_predicted_pwm, best_predicted_sites = 0, 0, None, []
  avg_runtime = 0.0

  for i in range(200):
    run_time, predicted_pwm, predicted_sites = motif_finding(dataset_path)
    avg_runtime += run_time
    score = calculate_icpc(predicted_pwm)

    if(score > best_score):
      print(f'dataset: {dataset_path.split("/")[2]}  ITER: {i} Score: {score}')
      best_run_time, best_predicted_pwm, best_predicted_sites = run_time, predicted_pwm, predicted_sites
      best_score = score

  print(f'dataset: {dataset_path.split("/")[2]} Best score: {best_score}')
  avg_runtime /= 200
  store_motif_finding_results(dataset_path, best_predicted_sites, best_predicted_pwm, avg_runtime)

if __name__ == '__main__':
  process_list = []
  paths = sorted(os.listdir("./datasets"))
  for name in paths:
    dataset_path = "./datasets/" + name
    print(dataset_path)
    p = Process(target=proc, args=(dataset_path,))
    process_list.append(p)

    p.start()


  for p in process_list:
    p.join()
