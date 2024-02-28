import React from "react";
import ReactDOM from "react-dom";
import { Header } from "./components/Header";
import AppProviders from "./contexts/AppProviders";
import { Playground } from "./components/play/Playground";

import "./styles/global.css";

const App = () => {
  return (
    <AppProviders>
      <div className="flex flex-col pb-12">
        <Header />
        <Playground />
      </div>
    </AppProviders>
  );
};

ReactDOM.render(<App />, document.getElementById("root"));
