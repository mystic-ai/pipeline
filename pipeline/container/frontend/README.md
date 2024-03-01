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
For development however, you can use the dev server by running

```shell
npm run dev
```

in this directory (where `package.json` lives).
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

Hence, the pipeline container must be up and running in order for this to work.
