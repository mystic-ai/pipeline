import { type SVGProps } from "react";

export interface IconTypes extends SVGProps<SVGSVGElement> {
  size?: number;
  className?: string;
  id?: string;
}
