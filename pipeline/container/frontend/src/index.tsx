import React from "react";
import ReactDOM from "react-dom";
import { Header } from "./components/Header";
import "./styles/global.css";
import { Input } from "./components/ui/Inputs/Input";
import { Checkbox } from "./components/ui/Inputs/Checkbox";
import AppProviders from "./contexts/AppProviders";
import { DynamicRunInput } from "./components/play/DynamicRunInput";
import { FormProvider } from "react-hook-form";
import { DynamicFieldsForm } from "./components/play/DynamicFieldsForm";

const App = () => {
  // GET pipeline
  // React.useEffect(() => {
  //   fetch(`/v4/container/pipeline`)
  //     .then((res) => res.json())
  //     .then((data) => {
  //       console.log(data);
  //     })
  //     .catch((err) => console.log(err));
  // }, []);
  return (
    <AppProviders>
      <div className="flex flex-col pb-12">
        <Header />
        <DynamicFieldsForm
          onSubmitHandler={(inputs) => {
            console.log("submitted");
          }}
          pipelineInputIOVariables={[
            { run_io_type: "string", default: "Hey there" },
            { run_io_type: "integer", default: 10 },
          ]}
        />
      </div>
    </AppProviders>
  );
};

ReactDOM.render(<App />, document.getElementById("root"));
