import React, {PureComponent} from 'react';
import {Card, CardBody} from 'reactstrap';
import Timeline from 'react-visjs-timeline';
import {Profile} from '../../lib/profile/profile';
import moment from 'moment';
import {TraceSelection} from '../../lib/trace/selection';
import CardHeader from 'reactstrap/lib/CardHeader';
import {chain, addIndex} from 'ramda';
import {Kernel} from '../../lib/profile/kernel';

interface Props
{
    profile: Profile;
    selection: TraceSelection;
    selectTrace: (selection: TraceSelection) => void;
}

export class KernelTimeline extends PureComponent<Props>
{

    render()
    {
        const start = new Date(this.props.profile.run.start);
        const end = new Date(this.props.profile.run.end);

        const options = {
            width: '100%',
            minHeight: '120px',
            zoomMin: 10,
            zoomMax: 60000,
            format: {
                minorLabels: {
                    millisecond: 's.SS'
                }
            },
            showTooltips: true,
            start, end,
            min: start,
            max: end,
            dataAttributes: ['id']
        };
        const selection = this.props.selection === null ? [] :
            [this.makeId(this.props.selection.kernel, this.props.selection.trace)];

        return (
            <Card>
                <CardHeader>Select kernel to examine</CardHeader>
                <CardBody>
                    <Timeline
                        options={options}
                        items={this.createTimelineItems(this.props.profile)}
                        selectHandler={this.handleTraceSelect}
                        selection={selection} />
                </CardBody>
            </Card>
        );
    }

    createTimelineItems = (profile: Profile) =>
    {
        const indexedChain = addIndex(chain);

        return indexedChain((kernel: Kernel, kernelIndex: number) => {
            return kernel.traces.map((trace, index) => {
                const start = new Date(trace.start);
                const end = new Date(trace.end);
                const label = `${kernel.name} #${index}`;

                return {
                    id: this.makeId(kernelIndex, index),
                    start,
                    end,
                    content: label,
                    title: `${label}: ${moment(start).format('HH:mm:ss.SSS')} - ${moment(end).format('HH:mm:ss.SSS')}`
                };
            });
        }, profile.kernels);
    }

    makeId = (kernelIndex: number, traceIndex: number): string =>
    {
        return `${kernelIndex}-${traceIndex}`;
    }

    handleTraceSelect = (props: {items: string[]}) =>
    {
        const value = props.items.length === 0 ? null : {
            kernel: parseInt(props.items[0].split('-')[0], 10),
            trace: parseInt(props.items[0].split('-')[1], 10)
        };
        this.props.selectTrace(value);
    }
}
