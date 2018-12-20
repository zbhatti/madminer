from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def export_param_card(benchmark, parameters, param_card_template_file, mg_process_directory, param_card_filename=None):
    # Open parameter card template
    with open(param_card_template_file) as file:
        param_card = file.read()

    # Replace parameter values
    for parameter_name, parameter_value in six.iteritems(benchmark):
        parameter_lha_block = parameters[parameter_name][0]
        parameter_lha_id = parameters[parameter_name][1]

        # Transform parameters if needed
        parameter_transform = parameters[parameter_name][4]
        if parameter_transform is not None:
            variables = {"theta": parameter_value}
            parameter_value = eval(parameter_transform, variables)

        block_begin = param_card.lower().find(("Block " + parameter_lha_block).lower())
        if block_begin < 0:
            raise ValueError("Could not find block {0} in param_card template!".format(parameter_lha_block))

        block_end = param_card.lower().find("Block".lower(), block_begin + 20)
        if block_end < 0:
            block_end = len(param_card)

        block = param_card[block_begin:block_end].split("\n")
        changed_line = False
        for i, line in enumerate(block):
            comment_pos = line.find("#")
            if i >= 0:
                line = line[:comment_pos]
            line = line.strip()
            elements = line.split()
            if len(elements) >= 2:
                try:
                    if int(elements[0]) == parameter_lha_id:
                        block[i] = "    " + str(parameter_lha_id) + "    " + str(parameter_value) + "    # MadMiner"
                        changed_line = True
                        break
                except ValueError:
                    pass

        if not changed_line:
            raise ValueError("Could not find LHA ID {0} in param_card template!".format(parameter_lha_id))

        param_card = param_card[:block_begin] + "\n".join(block) + param_card[block_end:]

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


def export_run_card(template_filename, run_card_filename, systematics=None):
    # Open parameter card template
    with open(template_filename) as file:
        run_card_template = file.read()

    run_card_lines = run_card_template.split("\n")

    # Changes to be made
    settings = OrderedDict()
    settings["use_syst"] = "False"
    if systematics is not None:
        settings["use_syst"] = "True"
        settings["systematics_program"] = "systematics"
        settings["systematics_arguments"] = create_systematics_arguments(systematics)

    # Remove old entries
    for i, line in enumerate(run_card_lines):
        comment_pos = line.find("#")
        if i >= 0:
            line = line[:comment_pos]

        try:
            line_value, line_key = line.split("=")
        except ValueError:
            continue
        line_key = line_key.strip()

        if line_key in settings:
            del run_card_lines[i]
            break

    # Add new entries
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

    if "mu" in systematics:
        systematics_arguments.append("'--mur={}'".format(systematics["mu"]))
        systematics_arguments.append("'--muf={}'".format(systematics["mu"]))
        systematics_arguments.append("'--together=mur,muf'")

    elif "mur" in systematics:
        systematics_arguments.append("'--mur={}'".format(systematics["mur"]))

    elif "muf" in systematics:
        systematics_arguments.append("'--muf={}'".format(systematics["muf"]))

    if "pdf" in systematics:
        systematics_arguments.append("'--pdf={}'".format(systematics["pdf"]))

    if len(systematics_arguments) > 0:
        return "[" + ", ".join(systematics_arguments) + "]"

    return ""
