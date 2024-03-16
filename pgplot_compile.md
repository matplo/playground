# compiling pgplot

- install homebrew

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- brew install gcc xquartz autoconf automake

```
brew install gcc xquartz autoconf automake
```

- export PATH=/opt/homebrew/bin:$PATH

```
export PATH=/opt/homebrew/bin:$PATH
```

## then make sure g77 points to gfortran

```
ln -s /opt/homebrew/bin/gfortran /opt/homebrew/bin/g77
```

## also gcc to gcc-13 (from homebrew)

```
ln -s /opt/homebrew/bin/gcc-13 /opt/homebrew/bin/gcc
```

## in drivers.list uncomment server

- for example:

```
XWDRIV 1 /XWINDOW   Workstations running X Window System                C
```

## compile

```
# if not done already
cd pgplot
```

```
./makemake $PWD linux g77_gcc_aout
make
```

### important note

```
*** Finished compilation of PGPLOT ***

Note that if you plan to install PGPLOT in a different
directory than the current one, the following files will be
needed.

      libpgplot.a
      grfont.dat
      rgb.txt
      pgxwin_server

Also note that subsequent usage of PGPLOT programs requires that
the full path of the chosen installation directory be named in
an environment variable named PGPLOT_DIR.
```

# example command to comile code that uses pgplot

- the instructions below rely that PGPLOT_DIR is set - for example:

```
export PGPLOT_DIR=/usr/local/src/pgplot
```

## compile `example.f` in a single line

```
gfortran -w -ffixed-line-length-0 -o example example.f -L$PGPLOT_DIR  -lpgplot -L/usr/X11R6/lib -lX11 `$PGPLOT_DIR/cpg/libgcc_path.sh` -lgcc -lm -lc
```

## putting this in a makefile

- create a makefile with the following contents (remember the useful copy button to the right of the field):

### compile ONE target - named example - depends on example.f

- see https://github.com/matplo/playground/tree/main/pgplot for all the example makefiles (get one and rename to `makefile` for make without argument to work or run `makefile -f <makefile>`)

```
TARGET = example
# note: you can set the PGPLOT_DIR environment variable to the location of the PGPLOT library
# or you can set it here
# PGPLOT_DIR = /usr/local/pgplot

CC = gfortran
CFLAGS = -w -ffixed-line-length-0
LIBS = -L$(PGPLOT_DIR) -lpgplot -L/usr/X11R6/lib -lX11 `$(PGPLOT_DIR)/cpg/libgcc_path.sh` -lgcc -lm -lc
SRC = $(TARGET).f

$(TARGET):      $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	@rm -f $(TARGET)
```

- then run

```
make clean
make
```

### compile ONE target - named example1 and example2 - depends on example1.f and example2.f

```
TARGETS = example example2
# note: you can set the PGPLOT_DIR environment variable to the location of the PGPLOT library
# or you can set it here
# PGPLOT_DIR = /usr/local/pgplot

CC = gfortran
CFLAGS = -w -ffixed-line-length-0
LIBS = -L$(PGPLOT_DIR) -lpgplot -L/usr/X11R6/lib -lX11 `$(PGPLOT_DIR)/cpg/libgcc_path.sh` -lgcc -lm -lc

all: $(TARGETS)

$(TARGETS):
	$(CC) $(CFLAGS) -o $@ $@.f $(LIBS)

clean:
	@rm -f $(TARGETS)
```

### compile ALL *.f files in the directory (!)

```
# note: you can set the PGPLOT_DIR environment variable to the location of the PGPLOT library
# or you can set it here
# PGPLOT_DIR = /usr/local/pgplot

CC = gfortran
CFLAGS = -w -ffixed-line-length-0
LIBS = -L$(PGPLOT_DIR) -lpgplot -L/usr/X11R6/lib -lX11 `$(PGPLOT_DIR)/cpg/libgcc_path.sh` -lgcc -lm -lc

SRCS = $(wildcard *.f)
TARGETS = $(SRCS:.f=)

all: $(TARGETS)

$(TARGETS): %: %.f
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	@rm -f $(TARGETS)
```

- note: you can select what files to use using the wildcard statement - for example to compile only the example-starting files use `SRCS = $(wildcard example*.f)`
