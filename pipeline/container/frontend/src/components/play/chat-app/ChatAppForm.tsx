import React, { ReactNode, useState } from "react";

interface Props {
  children: ReactNode;
  handleSubmit: (question: string) => void;
  handleInputChange?: (value: string) => void;
}

export function ChatAppForm({
  children,
  handleSubmit,
  handleInputChange,
}: Props): JSX.Element {
  const [inputValue, setInputValue] = useState<string>("");

  function handleFormSubmit() {
    handleSubmit(inputValue);
    setInputValue("");
  }

  function handleChange(value: string) {
    setInputValue(value);
    handleInputChange && handleInputChange(value);
  }

  return (
    <form
      className="p-4 flex gap-3 border-t bg-gray-100 dark:bg-black border-gray-300 dark:border-gray-700"
      onSubmit={(e) => {
        e.preventDefault();
        handleFormSubmit();
      }}
    >
      <textarea
        className="input max-w-full flex-1 min-h-full"
        placeholder="Ask chat something"
        value={inputValue}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && inputValue) {
            event.preventDefault();
            handleFormSubmit();
          }
        }}
        name="question"
        onChange={(event) => handleChange(event.target.value)}
        aria-label="message to send"
      />

      {children}
    </form>
  );
}
