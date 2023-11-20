"""Tests for the PyTorch integration

Using a conftest copied from the main repo for now
"""
from pytest import fixture, mark

from examol.score.utils.multifi import collect_outputs
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe
from exaredox.score.pytorch import RedoxModelsScorer


class FakeRecipe(PropertyRecipe):
    pass


@fixture()
def recipe() -> PropertyRecipe:
    return FakeRecipe('test', 'fast')


@fixture()
def multifi_recipes(recipe) -> list[PropertyRecipe]:
    return [recipe, FakeRecipe(recipe.name, 'medium'), FakeRecipe(recipe.name, 'slow')]


@fixture()
def training_set(multifi_recipes) -> list[MoleculeRecord]:
    """Fake training set"""

    output = []
    for s, y in zip(['C', 'CC', 'CCC'], [1., 2., 3.]):
        record = MoleculeRecord.from_identifier(s)
        record.properties[multifi_recipes[0].name] = dict(
            (recipe.level, y + i) for i, recipe in enumerate(multifi_recipes)
        )
        output.append(record)
    return output


@fixture()
def scorer():
    return RedoxModelsScorer()


@fixture(params=['MPNN', 'EGNN'])
def model_kwargs(request) -> tuple[str, dict[str, object]]:
    return request.param, dict(hidden_dim=32)


def test_convert(training_set, scorer):
    converted = scorer.transform_inputs(training_set)
    assert converted[0]['num_nodes'] == 5


@mark.parametrize('multifi', [False, True])
def test_flow(training_set, scorer, model_kwargs, multifi_recipes, multifi):
    # Make the model updates
    model_obj = (model_kwargs, None)

    # Prepare messages
    model_msg = scorer.prepare_message(model_obj, training=True)
    inputs = scorer.transform_inputs(training_set)
    outputs = scorer.transform_outputs(training_set, multifi_recipes[-1])
    lower_fidelities = None
    if multifi:
        lower_fidelities = collect_outputs(training_set, multifi_recipes[:-1])

    # Run the training
    update_msg = scorer.retrain(model_msg, inputs, outputs.tolist(), lower_fidelities=lower_fidelities)

    # Run the update
    model_obj = scorer.update(model_obj, update_msg)
    assert model_obj[1] is update_msg

    # Run inference
    model_msg = scorer.prepare_message(model_obj, training=False)
    pred_y = scorer.score(model_msg, inputs, lower_fidelities=lower_fidelities)
    assert pred_y.shape == (len(training_set),)
