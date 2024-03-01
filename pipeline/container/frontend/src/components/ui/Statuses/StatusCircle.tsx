import React from "react";
export type Status = "active" | "inactive" | "expired" | "warning";

interface Props {
  status: Status;
}
export const StatusCircle = ({ status }: Props) => {
  let bgStyle = "bg-success-600";

  if (status === "warning") bgStyle = "bg-yellow-500";
  if (status === "inactive") bgStyle = "bg-gray-300";
  if (status === "expired") bgStyle = "bg-error-600";

  return (
    <div className="flex justify-start items-center h-5 pointer-events-none bg-amber">
      <div className={`rounded-full w-3 h-3 ${bgStyle}`} />
    </div>
  );
};
