// This is the root node for the App.
const App = () => {
  return (
    <div className="flex flex-col gap-3">
      <Inner title="Hello React!" />
      <Inner title="Hello from me too?" />
      <Image />
    </div>
  );
};

interface InnerProps {
  title: string;
}
const Inner = ({ title }: InnerProps) => {
  return <div>{title}</div>;
};

const Image = () => {
  return <img src="/static/image.jpeg" />;
};
