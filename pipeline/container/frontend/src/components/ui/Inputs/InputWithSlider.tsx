import { ChangeEvent, useEffect, useState } from "react";
import { Slider } from "./Slider";

export interface InputWithSliderProps {
  step?: number;
  min?: number;
  max?: number;
  defaultValue: number[];
  onChange?: (newValue: number) => void;
}

export function InputWithSlider({
  onChange,
  min,
  max,
  step,
  defaultValue,
  ...rest
}: InputWithSliderProps): JSX.Element {
  // State
  const [inputValue, setInputValue] = useState<number[]>([
    Number(defaultValue),
  ]);

  useEffect(() => {
    onChange && onChange(inputValue[0]);
  }, [inputValue]);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInputValue([Number(e.target.value)]);
  };

  return (
    <div className="flex gap-4">
      <div className="max-w-[5.3125rem]">
        <input
          type="number"
          value={inputValue[0]}
          onChange={handleInputChange}
          step={step || 1}
          min={min}
          max={max}
          className="input"
          {...rest}
        />
      </div>

      <Slider
        defaultValue={[Number(inputValue)]}
        min={min}
        max={max}
        step={step || 1}
        value={inputValue}
        onValueChange={(value) => {
          setInputValue(value);
        }}
      />
    </div>
  );
}
