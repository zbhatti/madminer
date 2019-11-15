from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def export_param_card(benchmark, parameters, param_card_template_file, mg_process_directory, param_card_filename=None):
    # Open parameter card template
    with open(param_card_template_file) as file:
        param_card = file.read()
    lines = param_card.splitlines()

    # Replace parameter values
    for parameter_name, parameter_value in six.iteritems(benchmark):
        parameter_lha_block = parameters[parameter_name][0]
        parameter_lha_id = parameters[parameter_name][1]

        # Transform parameters if needed
        parameter_transform = parameters[parameter_name][4]
        if parameter_transform is not None:
            variables = {"theta": parameter_value}
            parameter_value = eval(parameter_transform, variables)
        parameter_value = float(parameter_value)

        # Find entry
        current_block = None
        changed_line = False
        for i, line in enumerate(lines):

            # Remove comment
            try:
                line = line.split("#")[0]
            except:
                pass

            elements = line.split()

            # See if block begin
            if len(elements) == 2 and elements[0].lower() == "block":
                current_block = elements[1].lower()

            elif len(elements) == 2 and parameter_lha_block.lower() == current_block:
                try:
                    lha_id = int(elements[0])
                except ValueError:
                    continue

                if lha_id == parameter_lha_id:
                    lines[i] = "    " + str(parameter_lha_id) + "    " + str(parameter_value) + "    # MadMiner"
                    changed_line = True
                    break

            elif len(elements) == 3 and elements[0].lower() == parameter_lha_block.lower():
                try:
                    lha_id = int(elements[1])
                except ValueError:
                    continue

                current_block = None
                if lha_id == parameter_lha_id:
                    lines[i] = (
                        str(parameter_lha_block)
                        + "    "
                        + str(parameter_lha_id)
                        + "    "
                        + str(parameter_value)
                        + "    # MadMiner"
                    )
                    changed_line = True
                    break

        if not changed_line:
            raise ValueError("Could not find LHA ID {0} in param_card template!".format(parameter_lha_id))

        param_card = "\n".join(lines)

    # Output filename
    if param_card_filename is None:
        param_card_filename = mg_process_directory + "/Cards/param_card.dat"

    # Save param_card.dat
    with open(param_card_filename, "w") as file:
        file.write(param_card)


def export_reweight_card(sample_benchmark, benchmarks, parameters, mg_process_directory, reweight_card_filename=None):
    # Global setup
    lines = [
        "# Reweight card generated by MadMiner",
        "",
        "# Global setup",
        "change output default",
        "change helicity False",
    ]

    for benchmark_name, benchmark in six.iteritems(benchmarks):
        if benchmark_name == sample_benchmark:
            continue

        lines.append("")
        lines.append("# MadMiner benchmark " + benchmark_name)
        lines.append("launch --rwgt_name=" + benchmark_name)

        for parameter_name, parameter_value in six.iteritems(benchmark):
            parameter_lha_block = parameters[parameter_name][0]
            parameter_lha_id = parameters[parameter_name][1]

            # Transform parameters if needed
            parameter_transform = parameters[parameter_name][4]
            if parameter_transform is not None:
                variables = {"theta": parameter_value}
                parameter_value = eval(parameter_transform, variables)

            lines.append("  set {0} {1} {2}".format(parameter_lha_block, parameter_lha_id, parameter_value))

        lines.append("")

    reweight_card = "\n".join(lines)

    # Output filename
    if reweight_card_filename is None:
        reweight_card_filename = mg_process_directory + "/Cards/reweight_card.dat"

    # Save param_card.dat
    with open(reweight_card_filename, "w") as file:
        file.write(reweight_card)


def export_run_card(template_filename, run_card_filename, systematics=None, override_settings=None):
    # Open parameter card template
    with open(template_filename) as file:
        run_card_template = file.read()

    run_card_lines = run_card_template.split("\n")

    # Do we actually have to run MadGraph's systematics feature?
    run_systematics = False
    for value in six.itervalues(systematics):
        if value[0] in ["pdf", "scale"]:
            run_systematics = True

    # Lines to be removed
    entries_to_comment_out = [
        "use_syst",
        "systematics_program",
        "systematics_argument",
        "systematics_arguments",
        "sys_scalefact",
        "sys_alpsfact",
        "sys_matchscale",
        "sys_pdf",
    ]

    # Changes to be made
    settings = OrderedDict()
    settings["use_syst"] = "False"
    if run_systematics:
        settings["use_syst"] = "True"
        settings["systematics_program"] = "systematics"
        settings["systematics_arguments"] = create_systematics_arguments(systematics)

    # Remove old entries
    for i, line in enumerate(run_card_lines):
        line_content = line
        # Remove comments
        try:
            line_content = line_content.split("#")[0]
        except:
            pass
        try:
            line_content = line_content.split("!")[0]
        except:
            pass

        # Split at last equal sign
        elements = line_content.split("=")
        if len(elements) < 2:
            continue
        line_key = elements[-1].strip()

        if line_key in entries_to_comment_out:
            run_card_lines[i] = "# {} # Commented out by MadMiner".format(line)
            continue

        if line_key in override_settings:
            run_card_lines[i] = "{} = {} # Overriden by MadMiner".format(override_settings[line_key], line_key)
            continue

    # Add new entries - sytematics
    run_card_lines.append("")
    run_card_lines.append("#*********************************************************************")
    run_card_lines.append("# MadMiner systematics setup                                         *")
    run_card_lines.append("#*********************************************************************")
    for key, value in six.iteritems(settings):
        run_card_lines.append("{} = {}".format(value, key))
    run_card_lines.append("")

    # Write new run card
    new_run_card = "\n".join(run_card_lines)
    with open(run_card_filename, "w") as file:
        file.write(new_run_card)


def create_systematics_arguments(systematics):
    """ Put together systematics_arguments string for MadGraph run card """

    if systematics is None:
        return ""

    systematics_arguments = []

    mur_done = False
    muf_done = False
    pdf_done = False

    for value in six.itervalues(systematics):
        if value[0] == "scale" and value[1] == "mu":
            if mur_done or muf_done:
                raise ValueError("Multiple nuisance parameter for scale variation!")
            systematics_arguments.append("'--mur={}'".format(value[2]))
            systematics_arguments.append("'--muf={}'".format(value[2]))
            systematics_arguments.append("'--together=mur,muf'")
            systematics_arguments.append("'--dyn=-1'")
            mur_done = True
            muf_done = True
        elif value[0] == "scale" and value[1] == "mur":
            if mur_done:
                raise ValueError("Multiple nuisance parameter for mur variation!")
            systematics_arguments.append("'--mur={}'".format(value[2]))
            systematics_arguments.append("'--dyn=-1'")
            mur_done = True
        elif value[0] == "scale" and value[1] == "muf":
            if muf_done:
                raise ValueError("Multiple nuisance parameter for muf variation!")
            systematics_arguments.append("'--muf={}'".format(value[2]))
            systematics_arguments.append("'--dyn=-1'")
            muf_done = True
        elif value[0] == "pdf":
            if pdf_done:
                raise ValueError("Multiple nuisance parameter for PDF variation!")
            systematics_arguments.append("'--pdf={}'".format(value[1]))
            pdf_done = True

    if len(systematics_arguments) > 0:
        return "[" + ", ".join(systematics_arguments) + "]"

    return ""
