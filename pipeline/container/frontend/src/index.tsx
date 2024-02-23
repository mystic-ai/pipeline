import React from "react";
import ReactDOM from "react-dom";
import { Header } from "./components/Header";
import "./styles/global.css";

const App = () => {
  // GET pipeline
  React.useEffect(() => {
    fetch(`/v4/container/pipeline`)
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
      })
      .catch((err) => console.log(err));
  }, []);
  return <Header />;
};

ReactDOM.render(<App />, document.getElementById("root"));
