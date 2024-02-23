import React from "react";
import ReactDOM from "react-dom";
import { Header } from "./components/Header";
import "./styles/global.css";

const App = () => <Header />;

ReactDOM.render(<App />, document.getElementById("root"));
