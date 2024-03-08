import React from "react";
import { type IconTypes } from "../Icons/types";

export function CurlLogo({ size = 24 }: IconTypes): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      fill="none"
      viewBox="0 0 13 10"
    >
      <path
        fill="#37A095"
        className={"stroke-0"}
        d="M12.041 1.364c-.289 0-.523-.216-.523-.482S11.752.4 12.04.4c.29 0 .524.216.524.482s-.235.482-.524.482zM6.841 9.6c-.289 0-.523-.216-.523-.482 0-.267.234-.482.524-.482.289 0 .523.215.523.482 0 .266-.234.482-.523.482zm5.2-9.6c-.529 0-.958.395-.958.882 0 .104.028.201.064.294L6.651 8.27c-.435.083-.768.422-.768.847 0 .487.43.882.959.882.529 0 .958-.395.958-.882a.807.807 0 00-.06-.276l4.519-7.118c.42-.093.74-.425.74-.842C13 .395 12.572 0 12.042 0z"
      ></path>
      <path
        fill="#1A7CB8"
        className={"stroke-0"}
        d="M8.12 1.364c-.29 0-.524-.216-.524-.482S7.83.4 8.12.4c.289 0 .523.216.523.482s-.234.482-.523.482zM2.92 9.6c-.29 0-.524-.216-.524-.482 0-.267.235-.482.524-.482.29 0 .523.215.523.482 0 .266-.234.482-.523.482zM8.12 0c-.53 0-.959.395-.959.882 0 .104.029.201.065.294L2.729 8.27c-.434.083-.768.422-.768.847 0 .487.43.882.959.882.53 0 .958-.395.958-.882a.807.807 0 00-.06-.276l4.52-7.118c.42-.093.74-.425.74-.842C9.078.395 8.648 0 8.12 0zM.958 2.838c.29 0 .524.216.524.482s-.235.482-.524.482c-.289 0-.523-.216-.523-.482s.234-.482.523-.482zm0 1.364c.53 0 .959-.395.959-.882a.81.81 0 00-.06-.276.943.943 0 00-.899-.607c-.067 0-.127.024-.19.036C.333 2.557 0 2.895 0 3.32c0 .487.43.882.958.882zM.435 6.406c0-.267.234-.482.523-.482.29 0 .524.215.524.482 0 .266-.235.481-.524.481-.289 0-.523-.215-.523-.481zm1.482 0a.807.807 0 00-.06-.276.943.943 0 00-.899-.607c-.067 0-.127.024-.19.036C.333 5.643 0 5.98 0 6.406c0 .487.43.882.958.882.53 0 .959-.395.959-.882z"
      ></path>
    </svg>
  );
}