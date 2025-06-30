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
    freetype-dev

# Copy package files first for better Docker layer caching
COPY package.json package-lock.json ./

# Install dependencies (including dev dependencies)
RUN npm ci

# Copy source code and tests
COPY src/ ./src/
COPY test/ ./test/
COPY public/ ./public/
COPY index.html vite.config.js eslint.config.js ./

# Expose port for development server
EXPOSE 3000

# Default command for development/testing
CMD ["npm", "run", "dev"]
