import chroma from 'chroma-js';
import {Badge as BsBadge} from 'reactstrap';
import styled from 'styled-components';

const Badge = styled(BsBadge)`
  margin-right: 5px;
  font-size: 12px;
`;

export const READ_COLOR = chroma('#24AE5D');
export const WRITE_COLOR = chroma('#BB1E3E');

export function getIdentifier(index: number): string
{
    if (index >= 26) return (index - 26).toString();
    return String.fromCharCode('A'.charCodeAt(0) + index);
}
