import React, {PureComponent} from 'react';
import moment from 'moment';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import styled from 'styled-components';

interface Props
{
    kernel: Kernel;
    trace: Trace;
}

const Wrapper = styled.div`

`;
const Row = styled.div`
  display: flex;
`;
const KernelName = styled.span`
  font-weight: bold;
`;
const Label = styled.div`
  min-width: 80px;
  margin-right: 10px;
`;

export class TraceHeader extends PureComponent<Props>
{
    render()
    {
        const start = moment(this.props.trace.start).format('HH:mm:ss.SSS');
        const end = moment(this.props.trace.end).format('HH:mm:ss.SSS');

        return (
            <Wrapper>
                <Row>
                    <Label>Kernel:</Label>
                    <KernelName>{this.props.kernel.name}</KernelName>
                </Row>
                <Row>
                    <Label>Duration:</Label>
                    <div>{start} - {end}</div>
                </Row>
            </Wrapper>
        );
    }
}
