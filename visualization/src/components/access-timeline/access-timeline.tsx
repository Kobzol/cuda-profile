import React, {PureComponent} from 'react';
import {Panel} from 'react-bootstrap';
import Timeline from 'react-visjs-timeline';
import {AccessType, Trace} from '../../lib/trace/trace';
import {Kernel} from '../../lib/trace/kernel';

interface Props
{
    kernel: Kernel;
    trace: Trace;
}

export class AccessTimeline extends PureComponent<Props>
{

    render()
    {
        const start = new Date(this.props.trace.start);
        const end = new Date(this.props.trace.end);

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
            <Panel className='trace' header={`Access timeline (select access)`}>
                <Timeline
                    options={options}
                    items={this.createTimelineItems(this.props.trace)}
                    selectHandler={this.handleTraceSelect}
                />
            </Panel>
        );
    }

    createTimelineItems = (trace: Trace) =>
    {
        console.log(trace);
        return trace.accesses.slice(0, 1).map((access, index) => {
                const start = new Date(trace.start);
                const end = new Date(trace.end);
                const label = `${access.kind === AccessType.Read ? 'Read' : 'Write'} #${index}`;

                return {
                    id: `${index}`,
                    start,
                    end,
                    content: label,
                    title: `${label} of ${access.size} bytes at ${access.address}`
                };
        });
    }

    handleTraceSelect = (props: {items: string[]}) =>
    {

    }
}
