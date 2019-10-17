# DeepHiC
DeepHiC is an integrative approach combining artificial intelligence - deep learning with high throughput experimental evidence of chromatin interaction leads to prioritizing the functional variants in disease- and phenotype-related loci.

Input
The average length of the sequences of restriction fragments was 3,614 bp. We fixed the sequence length at 4,000 bp. For sequences that were longer than 4,000 bp, the same number of nucleotides were cropped at the 5’ and 3’ ends. For sequences that were shorter than 4,000 bp, both ends were padded with an equal number of Ns to form 4,000-bp sequences. 

Nucleotides A, T, C and G are encoded as [1,0,0,0], [0,1,0,0], [0,0,1,0] and [0,0,0,1]

For better input, we saved the matrix (4000*16) as a grey image (tiff format).
