import React from "react";
import ReactDOM from "react-dom";
import { Header } from "./components/Header";
import AppProviders from "./contexts/AppProviders";

import "./styles/global.css";
import PipelinePlayWrapper from "./components/play/PipelinePlayWrapper";

const App = () => {
  return (
    <AppProviders>
      <div className="grid grid-rows-[52px_1fr] h-screen">
        <Header />
        <div className="flex justify-center py-12">
          <PipelinePlayWrapper />
        </div>
      </div>
    </AppProviders>
  );
};

ReactDOM.render(<App />, document.getElementById("root"));
