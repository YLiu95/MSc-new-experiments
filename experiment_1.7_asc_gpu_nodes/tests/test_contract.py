from ranker.contract import tasks


def test_all_task_configurations_are_trainable():
    catalogue = {task.key for task in tasks()}
    assert len(catalogue) == 638820
    assert {(128, 7, 16), (256, 21, 64), (512, 90, 128)} <= catalogue