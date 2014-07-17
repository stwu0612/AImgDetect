CC = gcc  
CFLAG = -Wall -g -w -fPIC 

SRC = ./src
OBJ = ./obj

all: test_demo a

LIB_FLAG = -L/usr/local/lib -L/usr/lib -L./lib
INCLUDE = -I./include -I/usr/local/include/opencv2.1/
MY_LIB = -lm -lcv -lcxcore -lcvaux -lhighgui -lml
objects =  $(OBJ)/ColorLayout.o		\
	     $(OBJ)/EdgeHist.o		\
	     $(OBJ)/Process.o		\
	     $(OBJ)/ImageTypeAJudge.o		\
	     $(OBJ)/Demo_linux.o        \
     	    
                
test_demo: $(objects)
	@echo ----------------------------------------------
	$(CC) $(CFLAG) $(INCLUDE) $(LIB_FLAG) $(MY_LIB) -o Demo_linux $(objects)
	@echo ----------------------------------------------

$(objects): %.o:
	@echo ----------------------------------------------
	$(CC) -c $(CFLAG) $(INCLUDE) -o $@ $(SRC)/$(@F:.o=.cpp)

a:
	ar -rsv libImageTypeAJudge_2_0_0.a ./obj/ImageTypeAJudge.o ./obj/ColorLayout.o ./obj/EdgeHist.o ./obj/Process.o

.PHONY : clean
clean :
	rm -f Demo_linux ./obj/*


