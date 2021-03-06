import numpy as np
import os 

NUCLEOTIDE_INDEX_MAP = {0:'A', 1:'C', 2:'G', 3:'T'}
NUCLEOTIDES = list(NUCLEOTIDE_INDEX_MAP.values())
UNIFORM_NUCLEOTIDE_PROBS = [0.25, 0.25, 0.25, 0.25]

# 2)
def generate_random_nucleo_sequence(sc, sl):
  '''
  Generates SC random sequences (with uniform nucleotide frequencies). Each random sequence has length SL.
  '''
  sequences = []
  for _ in range(sc):
    sequence = ''.join(np.random.choice(NUCLEOTIDES, sl, replace= True, p = UNIFORM_NUCLEOTIDE_PROBS))
    sequences.append(sequence)

    assert len(sequence) == sl

  return sequences

# 3)
def generate_nucleo_distribution(icpc, sc):
  '''
  Generates an array of 4 values corresponding to (A, C, G, T) values in each position which has an information content of icpc and has a total of sc
  '''
  assert type(icpc) == int or type(icpc) == float

  row = np.zeros(4)

  random_index = np.random.choice([0, 1, 2, 3], size = 1, replace = False)
  if(icpc == 1):
    p = 0.8105
  elif(icpc == 1.5):
    p = 0.9245
  else:
    p = 1.0

  row[random_index] = p
  for i in range(len(row)):
    if(i != random_index):
      row[i] = (1-p)/3

  return row


def generate_random_motif_pwm(icpc, ml, sc):
  '''
  Generates a random motif (position weight matrix) of length ML, with total information content (as discussed in class) being ICPC * ML.
  SC = total number of sequences
  ICPC = Information content per column given by sum_nucleotides(Pnucleotide * log (Pnucleotide / P_random_nucleo))  .icpc belongs to [1, 1.5, 2]
  ML = motif length

  Returns:
  A PWM of motif of dimensions (ml, 4)

  NOTE: This PWM contains the probabilities
  '''
  motif_pwm = np.zeros((ml, 4))
  for i in range(ml):
    motif_pwm[i, :] = generate_nucleo_distribution(icpc, sc)

  return motif_pwm

# 4)
def create_ml_sequence_from_pwm(motif_pwm, ml):
  '''
  motif_pwm = this will be a matrix of shape ('ml', 4)
              Each row will have 4 values representing the counts of (A, C, G, T) which will sum to 'sc'

  Return:
  This method will return a sequence of length 'ml' in which each position will follow the distribution given by each corresponding row in motif_pwm
  '''
  sequence = ''
  for i in range(ml):
    distribution = motif_pwm[i]
    sequence += ''.join(np.random.choice(NUCLEOTIDES, size = 1, replace = True, p = distribution))

  return sequence

def generate_binding_sites(motif_pwm, ml, sc):
  binding_sites = []
  for i in range(sc):
    binding_sites.append(create_ml_sequence_from_pwm(motif_pwm, ml))
    
  return binding_sites


# 5)
def plant_binding_sites(random_nucleotide_sequences, binding_sites):
  '''
  random_nucleotide_sequences = contains 'sc' # of sequences each of length 'sl'
  binding_sites = contains 'sc' # of binding_sites each of length 'ml'

  Return:
  returns a list of size 'sc' containing the index where the binding_site sequence was overwritten
  '''
  new_nucleotides = []
  replacement_indices = []

  for i in range(len(random_nucleotide_sequences)):
    nucleotide_sequence = random_nucleotide_sequences[i]  
    binding_site = binding_sites[i] 

    assert len(nucleotide_sequence) > len(binding_site)

    random_index_to_overwrite = np.random.randint(0, len(nucleotide_sequence) - len(binding_site) + 1) # 3
    replacement_indices.append(random_index_to_overwrite)

    replaced_nucleotide = nucleotide_sequence[:random_index_to_overwrite] + binding_site 
    if(random_index_to_overwrite + len(binding_site) < len(nucleotide_sequence)):
      replaced_nucleotide += nucleotide_sequence[random_index_to_overwrite + len(binding_site): ]
    
    assert len(replaced_nucleotide) == len(nucleotide_sequence)
    new_nucleotides.append(replaced_nucleotide)
  
  return new_nucleotides, replacement_indices

# 6)
def write_sequences_to_fasta(new_nucleotides, filepath):
  with open(filepath, 'w+') as fp:
    doc_text = ''

    for i, sequence in enumerate(new_nucleotides):
      doc_text += '>SEQUENCE_{}\n'.format(str(i+1))
      doc_text += (sequence + '\n')
    fp.write(doc_text)

def build_benchmark_dataset(icpc, ml, sc, sl, dataset_folder = ''):
  '''
  icpc = A positive number called ICPC (???information content per column???)
  ml = A positive integer called ML (???motif length???)
  sl = A positive integer called SL (???sequence length???)
  sc = A positive integer called SC (???sequence count???)
  '''

  # 2) Generate SC random sequences (with uniform nucleotide frequencies). Each random sequence has length SL.
  random_nucleotide_sequences = generate_random_nucleo_sequence(sc, sl)

  # 3) Generate a random motif (position weight matrix) of length ML, with total information content (as discussed in class) being ICPC * ML.
  random_motif_pwm = generate_random_motif_pwm(icpc, ml, sc)

  # 4) Generate SC binding sites by sampling from this random motif
  binding_sites = generate_binding_sites(random_motif_pwm, ml, sc)

  # 5) ???Plant??? one sampled site at a random location in each random sequence generated in step 2. ???Planting??? a site means overwriting the substring at that location with the site.
  new_nucleotides, binding_site_plant_indices = plant_binding_sites(random_nucleotide_sequences, binding_sites)

  # 6) Write out the SC sequences into a FASTA format file called ???sequences.fa??? (Search the web for information on this file format.)
  write_sequences_to_fasta(new_nucleotides, dataset_folder + '/sequences.fa')
  
  # 7) In a separate text file (called ???sites.txt???) write down the location of the planted site in each sequence. (Use any format, your code will be reading this file later.)
  with open(dataset_folder + '/sites.txt', 'w+') as fp:
    fp.writelines([str(index) + '\n' for index in binding_site_plant_indices])

  # 8) In a separate text file (called ???motif.txt???) write down the motif that was generated in step 3.
  with open(dataset_folder + '/motif.txt', 'w+') as fp:
    doc_text = '>MOTIF_1\t' + str(ml) + '\n'
    for i in range(ml):
      doc_text += '\t'.join([str(val) for val in random_motif_pwm[i]]) + '\n'
    
    doc_text += '<'
    fp.write(doc_text)

  # 9) In a separate test file (called ???motiflength.txt???) write down the motif length.
  with open(dataset_folder + '/motiflength.txt', 'w+') as fp:
    fp.write(str(ml))

"""### Running the benchmark generation"""
if __name__ == '__main__':
  # format: (icpc, ml, sc, sl)
  parameters_combos = [(2, 8, 10, 500), (1, 8, 10, 500), (1.5, 8, 10, 500), # icpc changes 
                      (2, 6, 10, 500), (2, 7, 10, 500), # ml changes
                      (2, 8, 5, 500), (2, 8, 20, 500)] # sc changes
  iterations_per_combo = 10

  for combo in parameters_combos:
    icpc, ml, sc, sl = combo
    for i in range(1, iterations_per_combo+1):
      foldername = './datasets/dataset_{}_{}_{}_{}_{}'.format(icpc, ml, sc, sl, i)
      os.makedirs(foldername)
      build_benchmark_dataset(icpc=icpc, ml=ml, sc=sc, sl=sl, dataset_folder = foldername)
    print('Combo {} done!'.format(i))