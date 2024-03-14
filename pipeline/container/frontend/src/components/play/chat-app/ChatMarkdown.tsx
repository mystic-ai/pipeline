import React from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import { MarkdownComponents } from "../../ui/MarkdownComponents";

export function ChatMarkdown({ content }: { content?: string }): JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <ReactMarkdown
        rehypePlugins={[rehypeRaw]}
        remarkPlugins={[remarkGfm]}
        children={content}
        components={MarkdownComponents}
      />
    </div>
  );
}
