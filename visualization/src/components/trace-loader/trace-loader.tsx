import React, {ChangeEvent, DragEvent, PureComponent} from 'react';
import {connect} from 'react-redux';
import {loadFile} from '../../lib/file-load/actions';
import {FileType, TraceFile} from '../../lib/file-load/file';
import {loadingFiles, validTraceFiles} from '../../lib/file-load/reducer';
import {GlobalState} from '../../lib/state/reducers';
import {Button, ListGroup, ListGroupItem, CardHeader, CardBody, Card, Progress, Badge, Input} from 'reactstrap';
import {buildProfile} from '../../lib/profile/actions';
import {Errors} from '../../lib/state/errors';
import {withRouter} from 'react-router';
import styled from 'styled-components';
import Alert from 'reactstrap/lib/Alert';

interface StateProps
{
    files: TraceFile[];
    validTraceFiles: TraceFile[];
    loadingFiles: TraceFile[];
    buildError: string;
}

interface DispatchProps
{
    loadFile: (file: File) => void;
    buildProfile: (files: TraceFile[]) => void;
}

const FileInput = styled(Input)`
  margin-bottom: 20px;
`;
const FileList = styled(ListGroup)`
  margin-bottom: 20px;
`;
const BuildError = styled(Alert)`
  margin-top: 10px;
`;

class TraceLoaderComponent extends PureComponent<StateProps & DispatchProps>
{
    render()
    {
        return (
            <div>
                <Card>
                    <CardHeader>Select recorded trace files</CardHeader>
                    <CardBody>
                        <div>
                            <FileInput
                                   type='file' multiple
                                   onChange={this.handleTraceChange}
                                   onDrop={this.handleTraceDrop} />
                        </div>
                        <FileList>
                            {this.props.files.map(this.renderFile)}
                        </FileList>
                        <Button
                            disabled={!this.canBuildProfile()}
                            onClick={this.buildProfile}
                            title='Show trace'
                            color='primary'>Show trace</Button>
                        {this.props.buildError && <BuildError color='danger'>{this.props.buildError}</BuildError>}
                    </CardBody>
                </Card>
            </div>
        );
    }

    renderFile = (file: TraceFile): JSX.Element =>
    {
        return (
            <ListGroupItem key={file.name}>
                {file.name} {this.renderFileLabel(file)}
            </ListGroupItem>
        );
    }
    renderFileLabel = (file: TraceFile): JSX.Element =>
    {
        if (file.loading) return this.renderFileProgress();
        if (file.error === Errors.None) return this.renderFileSuccess(file);
        return this.renderFileError(file);
    }
    renderFileProgress = (): JSX.Element =>
    {
        return <Progress striped value={100} />;
    }
    renderFileError = (file: TraceFile): JSX.Element =>
    {
        return <Badge color='danger'>{this.formatError(file.error)}</Badge>;
    }
    renderFileSuccess = (file: TraceFile): JSX.Element =>
    {
        return <Badge color='success'>Loaded ({this.formatFileType(file.type)})</Badge>;
    }

    handleTraceChange = (event: ChangeEvent<HTMLInputElement>) =>
    {
        for (let i = 0; i < event.target.files.length; i++)
        {
            this.props.loadFile(event.target.files[i]);
        }
    }
    handleTraceDrop = (event: DragEvent<HTMLInputElement>) =>
    {
        const files = event.dataTransfer.files;
        for (let i = 0; i < files.length; i++)
        {
            this.props.loadFile(files[i]);
        }
    }

    canBuildProfile = (): boolean =>
    {
        return this.props.validTraceFiles.length > 0 && this.props.loadingFiles.length < 1;
    }
    buildProfile = () =>
    {
        this.props.buildProfile(this.props.validTraceFiles);
    }

    formatError = (error: number): string =>
    {
        switch (error)
        {
            case Errors.InvalidFileContent: return 'Invalid file content';
            case Errors.InvalidFileFormat: return 'Invalid file format';
            default: return 'Unknown error';
        }
    }
    formatFileType = (type: FileType): string =>
    {
        switch (type)
        {
            case FileType.Metadata: return 'metadata file';
            case FileType.Run: return 'run file';
            case FileType.Trace: return 'trace file';
            default: return 'unknown file';
        }
    }
}

export const TraceLoader = withRouter(connect<StateProps, DispatchProps>((state: GlobalState) => ({
    files: state.fileLoader.files,
    buildError: state.profile.buildError,
    validTraceFiles: validTraceFiles(state),
    loadingFiles: loadingFiles(state)
}), ({
    loadFile: loadFile.started,
    buildProfile: buildProfile.started
}))(TraceLoaderComponent));
