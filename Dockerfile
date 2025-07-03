FROM node:22.14.0-alpine3.21@sha256:9bef0ef1e268f60627da9ba7d7605e8831d5b56ad07487d24d1aa386336d1944

WORKDIR /app

# Install build dependencies for canvas and other native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    cairo-dev \
    jpeg-dev \
    pango-dev \
    musl-dev \
    giflib-dev \
    pixman-dev \
    pangomm-dev \
    libjpeg-turbo-dev \
    freetype-dev \
    bash

# Copy package files first for better Docker layer caching
COPY package.json package-lock.json ./

# Clean install to avoid rollup issues
RUN rm -rf node_modules package-lock.json && npm install

# Copy source code and tests
COPY src/ ./src/
COPY test/ ./test/
COPY public/ ./public/
COPY index.html vite.config.js eslint.config.js ./
COPY run_tests.sh ./

# Make run_tests.sh executable
RUN chmod +x run_tests.sh

# Expose port for development server
EXPOSE 3000

# Set entrypoint to bash
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "bash"]
