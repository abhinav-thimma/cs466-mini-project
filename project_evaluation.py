import numpy as np
import os 

def read_motif_file(filepath):
  with open(filepath, 'r') as fp:
    lines = fp.readlines()
    ml = int(lines[0].split('\t')[1].strip())
    motif = np.zeros((ml, 4))

    for i, line in enumerate(lines[1:-1]):
      motif[i] = np.array([float(val.strip()) for val in line.split('\t')])
    
    return motif

def calculate_icpc(dataset_path):
  predicted_motif = read_motif_file(dataset_path + '/predictedmotif.txt')
  s = 0
  for row in predicted_motif:
    for nucleotide_prob in row:
      if(nucleotide_prob != 0):
        s += nucleotide_prob * np.log2(nucleotide_prob / 0.25)

  return s/ len(predicted_motif)

def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    eps = 1e-9
    a = a + eps
    a = a / np.sum(a) 
    b = b + eps
    b = b / np.sum(b)

    val = 0.0
    for i in range(len(a)):
      if(a[i] != 0 and b[i] != 0):
        val += a[i] * np.log(a[i]/ b[i])
    return val

def relative_entropy(dataset_path): 
  motif = read_motif_file(dataset_path + '/motif.txt')  # contains probabilities
  predicted_motif = read_motif_file(dataset_path + '/predictedmotif.txt') # contains probabilities
  
  relative_entropy = 0.0
  for i in range(len(motif)):
    relative_entropy += KL(predicted_motif[i], motif[i])
  
  return relative_entropy

def number_of_overlaps(dataset_path):
  original_sites = []
  with open(dataset_path + '/sites.txt', 'r') as fp:
    original_sites = [int(num.strip()) for num in fp.read().split()]

  predicted_sites = []
  with open(dataset_path + '/predictedsites.txt', 'r') as fp:
    predicted_sites = [int(num.strip()) for num in fp.read().split()]

  matches = 0
  for i in range(len(original_sites)):
    if(original_sites[i] == predicted_sites[i]):
      matches += 1
  
  return matches

def compute_metrics(dataset_path):
  rel_entropy = relative_entropy(dataset_path)
  overlap_count = number_of_overlaps(dataset_path)
  icpc = calculate_icpc(dataset_path)

  with open(dataset_path + '/avgruntime.txt') as fp:
    run_time = float(fp.read().strip())

  return rel_entropy, overlap_count, icpc, run_time

if __name__ == '__main__':
  paths = sorted(os.listdir("./datasets"))

  with open('./results.csv', 'w+') as fp:
    lines = []
    lines.append('Dataset_name, Relative_entropy, Sequence_count, Overlap_count, Expected_ICPC, ICPC, Motif_length, Average_runtime (s)\n')
    for name in paths:
      val = name.split('_')
      expected_icpc, ml, sc = val[1], val[2], val[3]
      rel_entropy, overlap_count, icpc, run_time = compute_metrics('./datasets/' + name)
      print(f'Dataset: {name}, Relative entropy: {rel_entropy}, overlap_count: {overlap_count}, icpc = {icpc}, Runtime = {run_time}')

      lines.append(f'{name}, {rel_entropy}, {sc}, {overlap_count}, {expected_icpc}, {icpc}, {ml}, {run_time}\n')
    fp.writelines(lines)

