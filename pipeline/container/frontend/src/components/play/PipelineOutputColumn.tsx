import React from "react";
import { PropsWithChildren } from "react";
import { PipelinePlayColumn } from "./PipelinePlayColumn";
import { Tooltip } from "../ui/Tooltips/Tooltip";
import { IconInfoCircle } from "../ui/Icons/IconInfoCircle";
import { Select, SelectItem } from "../ui/Inputs/Select";
import { StreamingMode } from "../../types";
import { Label } from "../ui/Inputs/Label";

interface Props extends PropsWithChildren {
  isStreaming: boolean;
  onStreamingModeChange: (mode: StreamingMode) => void;
}
interface StreamingModeFormProps {
  onChange: (value: StreamingMode) => void;
}
const StreamingModeForm = ({ onChange }: StreamingModeFormProps) => {
  const options: StreamingMode[] = ["append", "replace"];

  return (
    <div className="flex items-center gap-3">
      <Label>Streaming mode</Label>
      <Select
        defaultValue={"append"}
        onValueChange={(value: StreamingMode) => onChange(value)}
        size={"xs"}
      >
        {options.map((o, index) => (
          <SelectItem value={o.valueOf()} key={o.valueOf()}>
            {o.valueOf()}
          </SelectItem>
        ))}
      </Select>
    </div>
  );
};
interface TitleWithStreamingInfoProps {
  onStreamingModeChange: (mode: StreamingMode) => void;
}

const TitleWithStreamingInfo = ({
  onStreamingModeChange,
}: TitleWithStreamingInfoProps) => {
  return (
    <div className="flex justify-between w-full">
      <div className="flex items-center gap-2">
        <div>Output</div>
        <Tooltip
          content={"Output is streamed"}
          contentProps={{ align: "start" }}
        >
          <div>
            <IconInfoCircle className="stroke-slate-500" size={16} />
          </div>
        </Tooltip>
      </div>
      <div>
        <StreamingModeForm onChange={onStreamingModeChange} />
      </div>
    </div>
  );
};
export default function PipelineOutputColumn({
  isStreaming,
  children,
  onStreamingModeChange,
}: Props): JSX.Element {
  const className = "vcol-col-not-first";
  if (!isStreaming)
    return (
      <PipelinePlayColumn title="Output" className={className}>
        {children}
      </PipelinePlayColumn>
    );
  return (
    <PipelinePlayColumn
      title={
        <TitleWithStreamingInfo onStreamingModeChange={onStreamingModeChange} />
      }
      className={className}
    >
      {children}
    </PipelinePlayColumn>
  );
}
