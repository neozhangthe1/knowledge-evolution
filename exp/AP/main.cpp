#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include "ap.h"
using namespace std;

// test
int main(int argc, char** argv)
{
  vector<string> words;
  int prefType = 1;

  char* graphFile = argv[1];
  char* wordFile = argv[2];
  prefType = atoi(argv[3]);

  char line[256];
  FILE* wordFin = fopen(wordFile,"r");
  while(fgets(line, 255, wordFin)){
    if(feof(wordFin)){
      break;
    }
    char* split = strchr(line, '\t');

    if(split){
      *split = 0;
      words.push_back(string(line));
    } else {
      printf("ERROR\n");
    }
  }
  fclose(wordFin);
  printf("Load words done\n");
  FILE* graphFin = fopen(graphFile, "r");
  FILE* fout = fopen("result","w");
  vector<int> examplar = affinityPropagation(graphFin, prefType);
  for (size_t i = 0; i < examplar.size(); ++i) {
    //printf("%d ", examplar[i]);
    fprintf(fout, "%s %s\n", words[i].c_str(), words[examplar[i]].c_str());
  }
  fclose(fout);
  fclose(graphFin);
  puts("");
  return 0;
}
