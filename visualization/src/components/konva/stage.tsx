import React, {PureComponent} from 'react';
import * as Konva from 'konva';

interface Props
{
    id: string;
    width: number;
    height: number;
}

interface State
{
    stage: Konva.Stage;
    layer: Konva.Layer;
}

export class Stage extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            stage: null,
            layer: null
        };
    }

    componentDidMount()
    {
        const stage = new Konva.Stage({
            container: this.props.id,
            width: this.props.width,
            height: this.props.height
        });
        const layer = new Konva.Layer();
        stage.add(layer);

        this.setState(() => ({
            stage, layer
        }));
    }

    render()
    {
        return (
            <div id={this.props.id}>
                {React.Children.map(this.props.children, (child: JSX.Element) => {
                    return React.cloneElement(child, { ctx: this.state.layer });
                })}
            </div>
        );
    }
}
