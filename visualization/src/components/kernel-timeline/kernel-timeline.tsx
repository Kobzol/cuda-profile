import React, {PureComponent} from 'react';
import './kernel-timeline.css';
import {Panel} from 'react-bootstrap';
import Timeline from 'react-visjs-timeline';
import {Profile} from '../../lib/trace/profile';
import * as moment from 'moment';
import {flatMap} from 'lodash';
import {TraceSelection} from '../../lib/trace/trace-selection';

interface Props
{
    profile: Profile;
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
            zoomMin: 100,
            zoomMax: 60000,
            format: {
                minorLabels: {
                    millisecond: 's.SS'
                }
            },
            showTooltips: true,
            start: start,
            end: end,
            min: start,
            max: end,
            dataAttributes: ['id']
        };
        return (
            <Panel className='kernel' header={`Timeline (select a trace)`}>
                <Timeline
                    options={options}
                    items={this.createTimelineItems(this.props.profile)}
                    selectHandler={this.handleTraceSelect}
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
                    id: `${kernelIndex}-${index}`,
                    start,
                    end,
                    content: label,
                    title: `${label}: ${moment(start).format('HH:mm:ss.SSS')} - ${moment(end).format('HH:mm:ss.SSS')}`
                };
            });
        });
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