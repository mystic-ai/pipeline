# Pipeline container frontend

This is a Single Page Application for the frontend of the pipeline container play page, where you can quickly try out your pipeline locally.
The build files, e.g `index.html` and `bundle.js`, are loaded on to the container and served from there, at `localhost:14300/play`.

# Tech Stack

- Language: Typescript
- UI Library: React
- Other: tailwindcss, webpack

# How to contribute

After building your pipeline container, using `pipeline container build`, spin up the container in debug mode and override the `pipeline` SDK package installed on the container with your local pipeline package (that you want to develop) by bind mounting it to the container as a docker volume:

```shell
pipeline container up -v "/path/to/your/pipeline_SDK:/usr/local/lib/python3.10/site-packages/pipeline" --debug
```

After running this command, the pipeline container play page should be available at `localhost:14300/play`.
This serves the bundled `html`, `js` and `css` located under the `static` directory.
For development however, a dev server is available to you.
First install all the node modules by running

```shell
npm install
```

in this directory (where `package.json` lives) and then

```shell
npm run dev
```

This will spin up a dev server at `localhost:1234` with hot-reloading and avoids having to re-build your application to see your changes appear.

In order for the dev server to fetch pipeline information from the container API, the following proxy is defined:

```json
proxy: [
      {
        context: ["/v4"],
        target: "http://localhost:14300",
      },
    ],
```

This is necessary for instance, to get the inputs variables to the pipeline, which need to be fetched from the container API.

Hence, all endpoints under `/v4` on the container API are also available to you on the dev server. Note that this means the pipeline container must be up and running in order for the dev server to work as expected.

When you are happy with all your changes, simply build the frontend application by running:

```shell
npm run build
```

This will generate new bundled files under the `static` directory.
You should then see those changes appear on the pipeline container play page at `localhost:14300/play`.
