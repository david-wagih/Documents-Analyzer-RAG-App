declare module "react-drag-drop-files" {
  import { FC } from "react";

  interface FileUploaderProps {
    handleChange: (file: File) => void;
    onTypeError?: (err: Error) => void;
    types: string[];
    classes?: string;
    children: React.ReactNode;
  }

  export const FileUploader: FC<FileUploaderProps>;
}
