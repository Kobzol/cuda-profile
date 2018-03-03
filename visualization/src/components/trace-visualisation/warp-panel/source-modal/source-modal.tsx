import React, {PureComponent} from 'react';
import Modal from 'reactstrap/lib/Modal';
import ModalHeader from 'reactstrap/lib/ModalHeader';
import ModalBody from 'reactstrap/lib/ModalBody';
import {SourceView} from './source-view/source-view';
import {Trace} from '../../../../lib/profile/trace';
import {Kernel} from '../../../../lib/profile/kernel';
import {SourceLocation} from '../../../../lib/profile/metadata';
import styled from 'styled-components';

interface Props
{
    opened: boolean;
    kernel: Kernel;
    trace: Trace;
    locationFilter: SourceLocation[];
    onClose(): void;
    setLocationFilter(lines: SourceLocation[]): void;
}

const Body = styled(ModalBody)`
  padding: 0;
`;

export class SourceModal extends PureComponent<Props>
{
    render()
    {
        return (
            <Modal isOpen={this.props.opened}
                   toggle={this.props.onClose}>
                <ModalHeader toggle={this.props.onClose}>
                    {this.getFilename(this.props.kernel.metadata.source.file)}
                </ModalHeader>
                <Body>
                    <SourceView content={this.props.kernel.metadata.source.content}
                                file={this.props.kernel.metadata.source.file}
                                warps={this.props.trace.warps}
                                locationFilter={this.props.locationFilter}
                                setLocationFilter={this.props.setLocationFilter} />
                </Body>
            </Modal>
        );
    }

    getFilename = (file: string): string =>
    {
        const slash = file.lastIndexOf('/');
        if (slash === -1) return file;
        return file.substr(slash + 1);
    }
}
