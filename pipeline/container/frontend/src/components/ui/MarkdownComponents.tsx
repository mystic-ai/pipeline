import React from "react";
import type { Components } from "react-markdown";
import SyntaxHighlighter from "react-syntax-highlighter";
import lightStyle from "react-syntax-highlighter/dist/cjs/styles/hljs/a11y-light";
import { syntaxHighlighterStyleNightOwl } from "../ui/Code/syntaxHighlighterStyleNightOwl";

export const MarkdownComponents: Components = {
  h1: ({ node, ...props }) => (
    <h1 className="text-display_xs font-bold mb-3" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h2 className="text-xl font-bold mb-3" {...props} />
  ),
  h3: ({ node, ...props }) => (
    <h3 className="text-lg font-bold mb-3" {...props} />
  ),
  h4: ({ node, ...props }) => (
    <h4 className="text-base font-bold mb-3" {...props} />
  ),
  h5: ({ node, ...props }) => (
    <h5 className="text-sm font-bold mb-3" {...props} />
  ),
  h6: ({ node, ...props }) => (
    <h6 className="text-xs font-bold mb-3" {...props} />
  ),
  p: ({ node, ...props }) => (
    <p className="text-sm font-normal mb-1" {...props} />
  ),
  a: ({ node, ...props }) => (
    <a
      className="text-sm font-normal text-blue-500 hover:underline"
      {...props}
    />
  ),
  ul: ({ node, ...props }) => (
    <ul
      className="text-sm font-normal mb-3 list-disc pl-4 [&_ol]:mb-0 [&_li]:mb-0"
      {...props}
    />
  ),
  ol: ({ node, ...props }) => (
    <ol
      className="text-sm font-normal mb-3 list-decimal pl-4 [&_ol]:mb-0 [&_li]:mb-0"
      {...props}
    />
  ),
  li: ({ node, ...props }) => <li className="text-sm font-normal" {...props} />,
  blockquote: ({ node, ...props }) => (
    <blockquote
      className="text-base font-normal border-l-4 border-gray-200 pl-4"
      {...props}
    />
  ),
  code: ({ node, inline, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || "");
    return !inline && match ? (
      // If code block
      <div className="code-wrap">
        <SyntaxHighlighter
          {...props}
          children={String(children).replace(/\n$/, "")}
          language={match[1]}
          //@ts-ignore
          style={syntaxHighlighterStyleNightOwl}
          wrapLongLines
          customStyle={{
            borderRadius: "5px",
            fontSize: "12px",
            padding: "10px",
            width: "100%",
          }}
          PreTag="div"
        />
      </div>
    ) : (
      // If inline code
      <code {...props} className={`bg-gray-900 text-white p-2 text-xs rounded`}>
        {children}
      </code>
    );
  },

  hr: ({ node, ...props }) => (
    <hr className="h-px bg-gray-200 mb-3" {...props} />
  ),
  table: ({ node, ...props }) => (
    <table
      className="border-collapse border border-gray-200 text-left"
      {...props}
    />
  ),
  thead: ({ node, ...props }) => (
    <thead className="border-collapse border border-gray-200" {...props} />
  ),
  tbody: ({ node, ...props }) => (
    <tbody className="border-collapse border border-gray-200" {...props} />
  ),
  tr: ({ node, ...props }) => (
    <tr className="border-collapse border border-gray-200" {...props} />
  ),
  th: ({ node, ...props }) => (
    <th className="border-collapse border border-gray-200" {...props} />
  ),
  td: ({ node, ...props }) => (
    <td className="border-collapse border border-gray-200" {...props} />
  ),
  img: ({ node, ...props }) => <img className="mb-3" {...props} />,
};
