FROM ubuntu:22.04

# Setup timezone.
RUN    echo "Etc/UTC" > /etc/timezone                       \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime    \
    && apt-get update                                       \
    && apt-get install -q -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install base dependencies.
RUN    apt-get update        \
    && apt-get install -y    \
        sudo                 \
        cmake                \
        build-essential      \
        ninja-build          \
        git                  \
        rsync                \
        gdb                  \
        gdbserver            \
    && apt-get autoremove -y \
    && apt-get clean -y      \
    && rm -rf /var/lib/apt/lists/*

# Install llvm 16.
RUN    git clone https://gist.github.com/96573409aee8d12951337621ef07b027.git /tmp/install-llvm \
    && chmod +x /tmp/install-llvm/install.sh                                                    \
    && /tmp/install-llvm/install.sh 16                                                          \
    && rm -rf /tmp/install-llvm

# Install DSO dependencies.
RUN    apt-get update        \
    && apt-get install -y    \
        libsuitesparse-dev   \
        libeigen3-dev        \
        libboost-all-dev     \
        libopencv-dev        \
        libzip-dev           \
    && apt-get autoremove -y \
    && apt-get clean -y      \
    && rm -rf /var/lib/apt/lists/*

# Install Pangolin for DSO.
RUN    apt-get update                            \
    && apt-get install -y                        \
        libgl1-mesa-dev                          \
        libglew-dev                              \
        pkg-config                               \
        libegl1-mesa-dev                         \
        libwayland-dev                           \
        libxkbcommon-dev                         \
        wayland-protocols                        \
    && git clone https://github.com/stevenlovegrove/Pangolin.git /tmp/pangolin \
    && cd /tmp/pangolin                          \
    && git checkout v0.6                         \
    && cmake -Bbuild                             \
             -S.                                 \
             -DCMAKE_INSTALL_PREFIX=/usr/local   \
    && cmake --build build --target install      \
    && apt-get autoremove -y                     \
    && apt-get clean -y                          \
    && rm -rf /var/lib/apt/lists/*

# Arguments.
ARG USERNAME=nonroot
ARG UID=1000
ARG GID=1000

# Environment variables.
ENV USERNAME=$USERNAME
ENV UID=$UID
ENV GID=$GID

# Create the user.
RUN    groupadd --gid $GID $USERNAME \
    && adduser                       \
        --disabled-password          \
        --disabled-login             \
        --gecos ""                   \
        --uid $UID                   \
        --gid $GID                   \
        $USERNAME                    \
    && usermod -aG sudo $USERNAME    \
    && groups $USERNAME              \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" | tee -a /etc/sudoers.d/$USERNAME

# Create /workspaces folder owned by nonroot.
RUN mkdir /workspaces && chown $USERNAME:$USERNAME /workspaces
VOLUME /workspaces
WORKDIR /workspaces

# Set the default user.
USER $USERNAME

# Install Zsh and Oh My Zsh.
RUN git clone https://gist.github.com/fe0d401310134bb6012beb3627c367ee.git /tmp/install-zsh \
    && sudo chmod +x /tmp/install-zsh/install.sh \
    && /tmp/install-zsh/install.sh               \
    && sudo rm -rf /tmp/install-zsh

# Start zsh shell when the container starts
ENTRYPOINT [ "zsh" ]
CMD ["-l"]
