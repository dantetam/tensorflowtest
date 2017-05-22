fname = "./text_corpus.txt"
outputfname = "./processed_text_corpus.txt"

def main():
  with open(fname) as f:
    content = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content] 
  
  with open(outputfname, 'w+') as the_file:
    for line in content:
      the_file.write(line + " ")

main()