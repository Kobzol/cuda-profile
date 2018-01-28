import React, {PureComponent} from 'react';
import {Panel} from 'react-bootstrap';
import Timeline from 'react-visjs-timeline';
import {Profile} from '../../../lib/profile/profile';
import * as moment from 'moment';
import {flatMap} from 'lodash';
import {TraceSelection} from '../../../lib/trace/selection';

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
            <Panel header='Kernel timeline' bsStyle='info'>
                <Timeline
                    options={options}
                    items={this.createTimelineItems(this.props.profile)}
                    selectHandler={this.handleTraceSelect}
                    selection={selection}
                />
            </Panel>
        );
    }

    createTimelineItems = (profile: Profile) =>
    {
        return flatMap(profile.kernels, (kernel, kernelIndex) => {
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
        });
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
