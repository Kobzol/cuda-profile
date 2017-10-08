import React, {PureComponent} from 'react';
import {MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {select} from 'd3';
import {zoom} from 'd3-zoom';
import * as d3 from 'd3';
import Grid from 'd3-v4-grid';
import {range} from 'd3-array';
import * as Konva from 'konva';

interface Props
{
    id: string;
    accessGroup: MemoryAccessGroup;
}

export class ThreadGrid extends PureComponent<Props>
{
    componentDidMount()
    {
        this.d3(range(32 * 32));
    }

    konva = (data: number[]) =>
    {
        const width = 1200;
        const height = 600;
        const stage = new Konva.Stage({
            container: this.props.id,
            width: width,
            height: height
        });
        const layer = new Konva.Layer();
        stage.add(layer);

        const grid = Grid() // create new grid layout
            .data(data)
            .padding([0.1, 0.1])
            .bands(true)
            .size([width, height]); // set size of container

        grid.layout();

        for (const d of grid.nodes())
        {
            layer.add(new Konva.Rect({
                x: d.x,
                y: d.y,
                width: grid.nodeSize()[0],
                height: grid.nodeSize()[1],
                fill: 'blue'
            }));
        }

        layer.draw();

        let mouseDown = false;

        stage.on('mousedown', () => mouseDown = true);
        stage.on('mouseup', () => mouseDown = false);
        //stage.on('mouseleave', () => mouseDown = false);
        stage.on('mousemove', (x: any) => {
            //console.log(x);
            //console.log(mouseDown);
            if (mouseDown)
            {
                stage.offset({
                    x: stage.offset().x + x.evt.movementX,
                    y: stage.offset().y + x.evt.movementY
                });
                stage.batchDraw();
            }
        });

        const scaleBy = 1.01;
        stage.on('wheel', (e: any) =>
        {
            e.evt.preventDefault();
            const oldScale = stage.scaleX();
            const mousePointTo = {
                x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
                y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
            };
            const newScale = e.evt.deltaY > 0 ? oldScale * scaleBy : oldScale / scaleBy;
            stage.scale({ x: newScale, y: newScale });
            const newPos = {
                x: -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
                y: -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale
            };
            stage.position(newPos);
            stage.batchDraw();
        });
    }

    d3 = (data: number[]) =>
    {
        const container = select('#thread-grid');

        const width = (container.node() as any).getBoundingClientRect().width;
        const height = (container.node() as any).getBoundingClientRect().height;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const grid = Grid() // create new grid layout
            .data(data)
            .padding([0.1, 0.1])
            .bands(true)
            .size([width, height]); // set size of container

        grid.layout();

        const nodeSize = grid.nodeSize();

        const g = svg.append('g');

        const z = zoom()
            .scaleExtent([1 / 2, 4])
            .translateExtent([
                [-nodeSize[0], -nodeSize[1]],
                [width + nodeSize[0], height + nodeSize[1]]]
            )
            .on('zoom', () => {
                g.attr('transform', d3.event.transform);
            });

        const selection = svg.append('rect')
            .attr('width', width)
            .attr('height', height)
            .style('fill', 'none')
            .style('pointer-events', 'all')
            .call(z);

        const example = g
                .selectAll('rect')
                .data(grid.nodes())
                .enter()
                .append('rect')
                .attr('x', (d: {x: number}) => d.x)
                .attr('y', (d: {y: number}) => d.y)
                .attr('width', nodeSize[0])
                .attr('height', nodeSize[1])
                .attr('fill', 'rgb(0,0,255)')
            /*.on('mouseover', (d, i: number, k) => {
                select(k[i]).attr('fill', 'rgb(255,0,0)');
            })*/;
    }

    render()
    {
        return (
            <div id={this.props.id} style={{'width': '100%', height: '600px'}} />
        );
    }
}
