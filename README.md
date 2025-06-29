# Docker Dev Env for JS

# Running tests

This command builds a docker image with the code of this repository and runs the repository's tests

```sh
./build_docker.sh my_app
docker run -t my_app ./run_tests.sh
```

```
[+] Building 0.1s (10/10) FINISHED                                                                   docker:default
 => [internal] load build definition from Dockerfile                                                           0.0s
 => => transferring dockerfile: 226B                                                                           0.0s
 => [internal] load metadata for docker.io/library/node:22.14.0-alpine3.21@sha256:9bef0ef1e268f60627da9ba7d76  0.0s
 => [internal] load .dockerignore                                                                              0.0s
 => => transferring context: 154B                                                                              0.0s
 => [1/5] FROM docker.io/library/node:22.14.0-alpine3.21@sha256:9bef0ef1e268f60627da9ba7d7605e8831d5b56ad0748  0.0s
 => [internal] load build context                                                                              0.0s
 => => transferring context: 1.07kB                                                                            0.0s
 => CACHED [2/5] WORKDIR /app                                                                                  0.0s
 => CACHED [3/5] COPY package.json package-lock.json .                                                         0.0s
 => CACHED [4/5] RUN npm install                                                                               0.0s
 => CACHED [5/5] COPY . .                                                                                      0.0s
 => exporting to image                                                                                         0.0s
 => => exporting layers                                                                                        0.0s
 => => writing image sha256:80007dbaeba9813527f4a4e663e6d773256f6e42f1b3c3fdf713fe45b4897c2f                   0.0s
 => => naming to docker.io/library/my_app                                                                      0.0s


> my-react-app@0.0.0 test
> vitest


 RUN  v3.1.1 /app

 ✓ src/App.test.jsx (2 tests) 176ms
 ✓ test/basic.test.js (3 tests) 6ms
 ✓ test/suite.test.js (3 tests) 7ms

 Test Files  3 passed (3)
      Tests  8 passed (8)
   Start at  22:08:27
   Duration  3.74s (transform 93ms, setup 361ms, collect 282ms, tests 190ms, environment 1.95s, prepare 392ms)
```

# Running a specific test

This example runs all tests matching the name "basic":

```sh
./build_docker.sh my_app
docker run -t my_app ./run_tests.sh basic
```


# Running a vite dev server

Run this command to enable hot reloading via docker.

```sh
./build_docker.sh my_app
docker run --network=host -v .:/app -it my_app npm exec vite dev --host
```
