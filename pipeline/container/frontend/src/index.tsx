import React from "react";
import ReactDOM from "react-dom";
import { Greeting } from "./components/Greeting";
import { Farewell } from "./components/Farewell";
import { IconLoading } from "./components/icons";
import { Button } from "./components/buttons/button";
import "./styles/global.css";

const App = () => (
  <div>
    <Greeting name="Word" />
    <Farewell name="World" />
    <button>Hey</button>
    <Button colorVariant="primary-animated">Hey</Button>
  </div>
);

ReactDOM.render(<App />, document.getElementById("root"));
