# DeepHiC
DeepHiC is an integrative approach combining artificial intelligence - deep learning with high throughput experimental evidence of chromatin interaction leads to prioritizing the functional variants in disease- and phenotype-related loci.

# Input
We compared models taking different bin sizes as input. The results showed that the model training on 40kb bin pairs performed best. 

As DNA is a double helix, both the forward sequence and the reverse sequence were considered. Nucleotides A, T, C and G are encoded as [1,0,0,0], [0,1,0,0], [0,0,1,0] and [0,0,0,1] The sequences of each fragment pair were merged and converted to a one-hot matrix with 10,000 rows for 10kb bins (40,000 rows for 40kb bins and 100,000 rows for 100kb) and 16 columns encoding 4 nucleotides. 

For better input, we saved the matrix (40000*16) as a grey image (png format). The well trained DeepHiC model can be found at https://drive.google.com/file/d/1OKXwq_1L5Sqs2ip1nrPVpXIfPB5Ws99A/view?usp=sharing.

The difference between the two predicted interaction probabilities (YA1 calculated using the reference allele A1 and YA2 calculated using the other allele A2) was used to assess the functional impact of the SNP which was defined as DeepHiC functional score.
