#include <stdio.h>

int cuModuleLoadDataEx	(
  void * 	module,
  const void * 	image,
  unsigned int 	numOptions,
  int * 	options,
  void ** 	optionValues	 
)	
{
  printf("image: %s\n",(char const*)image);

  char* errorLog;
  size_t errorLogSize = 0;
  for(int i = 0; i < numOptions; ++i)
  {
    printf("%d: %d %p\n",i,options[i],optionValues[i]);

    switch(options[i])
    {
      case 5:
        errorLog = (char*)optionValues[i];
        printf("  Error Log Ptr: %p\n",errorLog);
        break;
      case 6:
        errorLogSize = *((unsigned int*)optionValues[i]);
        printf("  Size: %ld\n",errorLogSize);
        break;
    }
  }


  strncpy(errorLog,"TEST\n",5);
  

  return 666;
}
