import logging

import pkg_resources
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/play", tags=["play"])


@router.get("", response_class=HTMLResponse)
async def render_pipeline_play():
    ts_code = pkg_resources.resource_string(
        "pipeline", "container/frontend/app.tsx"
    ).decode("utf-8")

    return f"""
    <!-- Your code that needs compiling goes in a type="text/babel" `script` tag -->
    <script type="text/babel" data-presets="react,stage-3">
    {ts_code}
    ReactDOM.render(<App />, document.getElementById("root"));
    </script>
    <div id="root"></div>
    <!-- This is what supports JSX compilation (and other transformations) -->
    <script src="https://unpkg.com/@babel/standalone@7.10.3/babel.min.js"></script>
    <link
        href="https://cdn.jsdelivr.net/npm/tailwindcss@2/dist/tailwind.min.css"
        rel="stylesheet"
    >
    <!-- These are for React -->
    <script
    src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.development.js"
    ></script>
    <script
    src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.development.js"
    ></script>
    """
