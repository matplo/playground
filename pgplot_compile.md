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

cd pgplot
./makemake $PWD linux g77_gcc_aout
# then make sure g77 points to gfortran

```
ln -s /opt/homebrew/bin/gfortran /opt/homebrew/bin/g77
```

# also gcc to gcc-13 (from homebrew)

ln -s /opt/homebrew/bin/gcc-13 /opt/homebrew/bin/gcc

# in drivers.list uncomment server

- for example:

```
XWDRIV 1 /XWINDOW   Workstations running X Window System                C
```

# compile

make

