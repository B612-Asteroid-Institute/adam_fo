#!/bin/bash
set -e  # Exit on error

# Determine the build directory location
if [ -z "$XDG_DATA_HOME" ]; then
    XDG_DATA_HOME="$HOME/.local/share"
fi
BUILD_DIR="$XDG_DATA_HOME/adam_fo"

echo "Installing find_orb dependencies in: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

#build and install find orb
mkdir -p find_orb
cd find_orb

# Function to clone and checkout a specific commit
clone_repo() {
    local repo=$1
    local commit=$2
    echo "Cloning $repo..."
    git clone "https://github.com/Bill-Gray/$repo.git"
    cd "$repo"
    git checkout "$commit"
    cd ..
}

# Clone all required repositories
clone_repo "lunar" "005ccaa469b32abc0df84b512d52e8b1c80efbda"
clone_repo "sat_code" "66afa0860da796a4ff4b4557531fb4e6ae44a095"
clone_repo "jpl_eph" "0c2782e86e42df69a55c2b2db0b72a40312c79c0"
clone_repo "find_orb" "f574af87ed11f5ec5b69ef8125e8b539de6d6645"
clone_repo "miscell" "f4565afdf9d0324e798527f837e0814f8de0abe0"

# Build and install components
echo "Building jpl_eph..."
cd jpl_eph
make libjpl.a
make install
cd ..

echo "Building lunar..."
cd lunar
make
make integrat
make install
cd ..

echo "Building sat_code..."
cd sat_code
make sat_id
make install
cd ..

echo "Building find_orb..."
cd find_orb
make
make install
cd ..

# Download ephemeris files
mkdir -p "$BUILD_DIR/find_orb/.find_orb"
cd "$BUILD_DIR/find_orb/.find_orb"

echo "Downloading ephemeris files..."
if [ ! -f linux_p1550p2650.440t ]; then
    wget https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440t/linux_p1550p2650.440t
fi

if [ ! -f bc405.dat ]; then
    wget -O bc405.dat https://storage.googleapis.com/asteroid-institute-data/ephemeris/bc405.dat
fi

echo "Installation complete! Build files are located in: $BUILD_DIR"
