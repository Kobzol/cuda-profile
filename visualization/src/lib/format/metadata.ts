export interface Metadata
{
    type: string;
    kernel: string;
    typeMap: string[];
    locations: DebugLocation[];
    source: SourceMetadata;
}

export interface DebugLocation
{
    name: string;
    file: string;
    line: number;
}

export interface SourceMetadata
{
    file: string;
    line: number;
    content: string;
}
