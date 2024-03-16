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

```
gfortran -w -ffixed-line-length-0 -o example example.f -L$PGPLOT_DIR  -lpgplot -L/usr/X11R6/lib -lX11 `$PGPLOT_DIR/cpg/libgcc_path.sh` -lgcc -lm -lc
```

- putting this in a makefile

```
example:	example.f
	gfortran -w -ffixed-line-length-0 -o example example.f -L$(PGPLOT_DIR)  -lpgplot -L/usr/X11R6/lib -lX11 `$(PGPLOT_DIR)/cpg/libgcc_path.sh` -lgcc -lm -lc

clean:
	rm -f example
```

- then run

```
make clean
make
```

