import React from "react";
import ReactDOM from "react-dom";
import { Greeting } from "./components/Greeting";
import { Farewell } from "./components/Farewell";

const App = () => (
  <div>
    <Greeting name="Word" />
    <Farewell name="World" />
    <button>Hey</button>
    <button>Hey</button>
  </div>
);

ReactDOM.render(<App />, document.getElementById("root"));
