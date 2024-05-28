#pragma once
#include <stdio.h>


#define check_ptr(val, name)  if (!val) {printf("null ptr %s", name); assert(0);}

