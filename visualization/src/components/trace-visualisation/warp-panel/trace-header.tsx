import React, {PureComponent} from 'react';
import moment from 'moment';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {Button} from 'reactstrap';
import styled from 'styled-components';
import {TraceSelection} from '../../../lib/trace/selection';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectTrace(selection: TraceSelection): void;
}

const Wrapper = styled.div`

`;
const KernelName = styled.span`
  font-weight: bold;
`;

export class TraceHeader extends PureComponent<Props>
{
    render()
    {
        const start = moment(this.props.trace.start).format('HH:mm:ss.SSS');
        const end = moment(this.props.trace.end).format('HH:mm:ss.SSS');

        return (
            <Wrapper>
                <div>
                    <KernelName>{this.props.kernel.name}</KernelName>
                    <span>{` from ${start} to ${end}`}</span>
                </div>
                <Button
                    onClick={this.deselectTrace}
                    color='primary' outline>
                    Select another kernel
                </Button>
            </Wrapper>
        );
    }



    deselectTrace = () =>
    {
        this.props.selectTrace(null);
    }
}
