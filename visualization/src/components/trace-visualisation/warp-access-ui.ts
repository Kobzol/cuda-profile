import chroma from 'chroma-js';
import {Badge as BsBadge} from 'reactstrap';
import styled from 'styled-components';

const Badge = styled(BsBadge)`
  margin-right: 5px;
  font-size: 12px;
`;

export const BadgeRead = styled(Badge)`
  background-color: #006400;
`;
export const BadgeWrite = styled(Badge)`
  background-color: #8B0000;
`;

export const WRITE_COLOR = chroma(180, 20, 0);
export const READ_COLOR = chroma(20, 180, 20);

export function getIdentifier(index: number): string
{
    if (index >= 26) return (index - 26).toString();
    return String.fromCharCode('A'.charCodeAt(0) + index);
}
