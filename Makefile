CC=g++ -std=c++0x -g
LINK= -Llib -lOPF $(shell pkg-config --libs opencv)
ODIR=./obj
OUT=./bin/comb2
SRC=./src
INC= $(patsubst %, -I%, $(shell find include -type d) $(shell find LibOPF/include -type d)) \
	$(shell pkg-config --cflags opencv)

_OBJ_DIR= $(shell find $(SRC)/ -type d)

OBJ_DIR = $(patsubst $(SRC)/%, $(ODIR)/%, $(_OBJ_DIR))

HEURISTIC=0

_OBJS = $(shell find  $(SRC)/ -name "*.cpp")

OBJS = $(patsubst $(SRC)/%.cpp, $(ODIR)/%.o, $(_OBJS))


$(ODIR)/opf_%.o: 

$(ODIR)/%.o: $(SRC)/%.cpp 
	@echo 'Compiling $<'
	@$(CC) -Wall -c $< $(INC) -o $@ -std=c++11

all: make_folders lib/libOPF.a $(OBJS)
	@echo Linking ...
	@$(CC) -o $(OUT) $(OBJS) $(LINK)

make_folders:
	@echo $(INC)
	@echo $(OBJS)
	@mkdir -p $(OBJ_DIR)

lib/libOPF.a: LibOPF/lib/libOPF.a
	@cp LibOPF/lib/libOPF.a lib/

LibOPF/lib/libOPF.a:
	@echo Compiling LibOPF
	@cd LibOPF ; make libOPF

run:
	$(OUT)

clean:
	rm -f $(OBJS) $(OUT)

zip:
	zip -r libMCS *